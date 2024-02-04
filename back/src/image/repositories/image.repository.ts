import { EntityManager } from "typeorm";
import { CreateImageDto } from '../dtos/CreateImage.dto';
import { Image } from "../entities/image.entity";
import { UpdateImageDto } from "../dtos/UpdateImage.dto";

export class ImageRepository {

      // save image
      async saveImage(image: Image, manager: EntityManager){
            try{
                   return await manager.createQueryBuilder(Image, 'image').insert().into('image').values(image).execute();
            }catch(err){
                  throw err;
            }     
      }

      // get image by imageId
      async getImageByImageId(imageId: string, manager: EntityManager){
            try{
                  return await manager.createQueryBuilder(Image, 'image').where('image_id = :imageId', {imageId}).getOne();
            }catch(err){
                  throw err;
            }
      }

      // update image by imageId
      async updateImage(imageId:string, toUpdate:UpdateImageDto, manager: EntityManager){
            try{
                  return await manager.createQueryBuilder(Image, 'image').update().set(toUpdate).where('image_id = :imageId', {imageId}).execute();
            }catch(err){
                  throw err;
            }
      }

      // delete image
      async deleteImage(imageId:string, manager: EntityManager){
            try{
                  return await manager.createQueryBuilder(Image, 'image').delete().where('image_id = :imageId', {imageId}).execute();
            }catch(err){
                  throw err;
            }
      }

}