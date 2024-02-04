import { Entity, PrimaryGeneratedColumn, Column, Timestamp } from "typeorm";
import { UpdateSplitImageDto } from "../dtos/UpdateSplitImage.dto";
import { CreateSplitImageDto } from "../dtos/CreateSplitImage.dto";
import { float } from "aws-sdk/clients/cloudfront";
import { double } from "aws-sdk/clients/lightsail";

@Entity()
export class SplitImage {
  @PrimaryGeneratedColumn()
  splitImageId: number;

  @Column()
  imageId: number;

  @Column()
  foodName: string;

  @Column({ type: 'real' })
  xCoordinate: number;

  @Column({ type: 'float' })
  yCoordinate: number;

  @Column({ type: 'float' })
  width: number;

  @Column({ type: 'float' })
  height: number;

  @Column({type: "timestamp with time zone", default: () => "CURRENT_TIMESTAMP"})
  createdAt: Timestamp;

  mapCreateSplitImageDtoToEntity(splitImageDto: CreateSplitImageDto  ) {
    const splitImage = new SplitImage();
    splitImage.imageId = splitImageDto.imageId;
    splitImage.xCoordinate = splitImageDto.xCoordinate;
    splitImage.yCoordinate = splitImageDto.yCoordinate;
    splitImage.width = splitImageDto.width;
    splitImage.height = splitImageDto.height;
    return splitImage;
  }

  mapUpdateSplitImageDtoToEntity(splitImageDto: UpdateSplitImageDto) {
    const splitImage = new SplitImage();
    splitImage.xCoordinate = splitImageDto.xCoordinate;
    splitImage.yCoordinate = splitImageDto.yCoordinate;
    splitImage.width = splitImageDto.width;
    splitImage.height = splitImageDto.height;
    return splitImage;
  }


}
